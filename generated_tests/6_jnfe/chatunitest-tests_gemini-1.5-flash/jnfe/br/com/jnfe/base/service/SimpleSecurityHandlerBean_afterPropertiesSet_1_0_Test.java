package br.com.jnfe.base.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import org.w3c.dom.Element;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class SimpleSecurityHandlerBean_afterPropertiesSet_1_0_Test {

    @Mock
    private Logger logger;

    @Test
    void testAfterPropertiesSet_nullAliasAndPassowrd() throws Exception {
        SimpleSecurityHandlerBean bean = new SimpleSecurityHandlerBean();
        // Inject the mock logger using reflection.  This is generally discouraged for production code, but acceptable in testing.
        Field loggerField = SimpleSecurityHandlerBean.class.getDeclaredField("logger");
        loggerField.setAccessible(true);
        loggerField.set(bean, logger);
        bean.afterPropertiesSet();
        verify(logger, times(2)).warn(anyString());
        assertEquals("teste", getAlias(bean));
        assertArrayEquals("teste".toCharArray(), getPassword(bean));
    }

    @Test
    void testAfterPropertiesSet_nullAlias() throws Exception {
        SimpleSecurityHandlerBean bean = new SimpleSecurityHandlerBean();
        bean.setPassword("password".toCharArray());
        Field loggerField = SimpleSecurityHandlerBean.class.getDeclaredField("logger");
        loggerField.setAccessible(true);
        loggerField.set(bean, logger);
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
        assertEquals("teste", getAlias(bean));
        assertArrayEquals("password".toCharArray(), getPassword(bean));
    }

    @Test
    void testAfterPropertiesSet_nullPassword() throws Exception {
        SimpleSecurityHandlerBean bean = new SimpleSecurityHandlerBean();
        bean.setAlias("alias");
        Field loggerField = SimpleSecurityHandlerBean.class.getDeclaredField("logger");
        loggerField.setAccessible(true);
        loggerField.set(bean, logger);
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
        assertEquals("alias", getAlias(bean));
        assertArrayEquals("teste".toCharArray(), getPassword(bean));
    }

    @Test
    void testAfterPropertiesSet_validAliasAndPassword() throws Exception {
        SimpleSecurityHandlerBean bean = new SimpleSecurityHandlerBean();
        bean.setAlias("alias");
        bean.setPassword("password".toCharArray());
        Field loggerField = SimpleSecurityHandlerBean.class.getDeclaredField("logger");
        loggerField.setAccessible(true);
        loggerField.set(bean, logger);
        bean.afterPropertiesSet();
        verify(logger, never()).warn(anyString());
        assertEquals("alias", getAlias(bean));
        assertArrayEquals("password".toCharArray(), getPassword(bean));
    }

    // Helper methods to access private fields for testing
    private String getAlias(SimpleSecurityHandlerBean bean) throws Exception {
        Field field = SimpleSecurityHandlerBean.class.getDeclaredField("alias");
        field.setAccessible(true);
        return (String) field.get(bean);
    }

    private char[] getPassword(SimpleSecurityHandlerBean bean) throws Exception {
        Field field = SimpleSecurityHandlerBean.class.getDeclaredField("password");
        field.setAccessible(true);
        return (char[]) field.get(bean);
    }
}
