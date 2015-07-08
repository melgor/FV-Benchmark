/* 
* @Author: melgor
* @Date:   2014-06-02 18:46:20
* @Last Modified 2015-03-03
*/

#include "Parser.hpp"
#include <boost/filesystem.hpp>
#include <iostream>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

Parser::Parser()
{
  desc.add_options()
    ("config", po::value<std::string>(&config)->default_value("")->required(), "path to config in .ini file")
    ("help", po::value<bool>(&help)->default_value(false)->zero_tokens(), "show help- this message")
    ;

}

void 
Parser::read(int argc, char** argv)
{
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if(help || argc < 2)
  {
    printHelp();
    exit(0);
  }

}

void 
Parser::printHelp()
{
  std::cout << desc << std::endl;
 
}

Parser::~Parser()
{

}