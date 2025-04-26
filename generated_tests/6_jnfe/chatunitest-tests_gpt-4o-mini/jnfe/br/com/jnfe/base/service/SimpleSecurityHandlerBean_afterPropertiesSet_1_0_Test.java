package br.com.jnfe.base.service;

import org.slf4j.Logger;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

public class SimpleSecurityHandlerBean_afterPropertiesSet_1_0_Test {

    private SimpleSecurityHandlerBean simpleSecurityHandlerBean;

    private Logger logger;

    @BeforeEach
    public void setUp() {
        simpleSecurityHandlerBean = new SimpleSecurityHandlerBean();
        logger = mock(Logger.class);
        // Using reflection to set the logger field in the SimpleSecurityHandlerBean
        try {
            java.lang.reflect.Field loggerField = SimpleSecurityHandlerBean.class.getDeclaredField("logger");
            loggerField.setAccessible(true);
            loggerField.set(simpleSecurityHandlerBean, logger);
        } catch (Exception e) {
            fail("Reflection setup failed: " + e.getMessage());
        }
    }

    @Test
    public void testAfterPropertiesSet_WithNullAliasAndPassword() throws Exception {
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Verify that the default values are set
        assertEquals("teste", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("teste".toCharArray(), getPrivateField(simpleSecurityHandlerBean, "password"));
        // Verify that the logger was called with the expected warnings
        verify(logger).warn("Utilzando apelido 'teste' para localizar chave particular no armaz�m de clientes");
        verify(logger).warn("Utilzando senha 'teste' para abrir chave particular no armaz�m de clientes");
    }

    @Test
    public void testAfterPropertiesSet_WithNonNullAliasAndNullPassword() throws Exception {
        simpleSecurityHandlerBean.setAlias("customAlias");
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Verify that the alias is set to the custom value and password is default
        assertEquals("customAlias", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("teste".toCharArray(), getPrivateField(simpleSecurityHandlerBean, "password"));
        // Verify that the logger was called for password only
        verify(logger).warn("Utilzando senha 'teste' para abrir chave particular no armaz�m de clientes");
    }

    @Test
    public void testAfterPropertiesSet_WithNullAliasAndNonNullPassword() throws Exception {
        simpleSecurityHandlerBean.setPassword("customPassword".toCharArray());
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Verify that the password is set to the custom value and alias is default
        assertEquals("teste", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("customPassword".toCharArray(), getPrivateField(simpleSecurityHandlerBean, "password"));
        // Verify that the logger was called for alias only
        verify(logger).warn("Utilzando apelido 'teste' para localizar chave particular no armaz�m de clientes");
    }

    @Test
    public void testAfterPropertiesSet_WithNonNullAliasAndNonNullPassword() throws Exception {
        simpleSecurityHandlerBean.setAlias("customAlias");
        simpleSecurityHandlerBean.setPassword("customPassword".toCharArray());
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Verify that the alias and password retain their custom values
        assertEquals("customAlias", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("customPassword".toCharArray(), getPrivateField(simpleSecurityHandlerBean, "password"));
        // Verify that the logger was not called
        verify(logger, never()).warn(anyString());
    }

    // Helper method to get private field values using reflection
    private <T> T getPrivateField(SimpleSecurityHandlerBean bean, String fieldName) {
        try {
            java.lang.reflect.Field field = SimpleSecurityHandlerBean.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (T) field.get(bean);
        } catch (Exception e) {
            fail("Reflection failed: " + e.getMessage());
            // Unreachable code
            return null;
        }
    }
}
