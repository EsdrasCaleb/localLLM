package br.com.jnfe.base.service;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Pkcs12SecurityHandlerBean_handle_0_0_Test.class, Pkcs12SecurityHandlerBean_afterPropertiesSet_3_0_Test.class, Pkcs12SecurityHandlerBean_loadKeyStore_4_0_Test.class })
public class Pkcs12SecurityHandlerBean_Suite {
}
