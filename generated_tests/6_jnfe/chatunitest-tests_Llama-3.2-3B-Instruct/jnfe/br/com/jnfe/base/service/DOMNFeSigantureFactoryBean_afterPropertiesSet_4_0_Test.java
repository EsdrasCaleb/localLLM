// Test method
package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_0_Test {

    private static final String PROVIDER_NAME = "testProvider";

    private static final String PROVIDER_CLASS_NAME = "testClass";

    private static final String DEFAULT_PROVIDER_NAME = "defaultProvider";

    private static final String DEFAULT_PROVIDER_CLASS_NAME = "defaultClass";

    @InjectMocks
    private DOMNFeSigantureFactoryBean bean;

    @BeforeAll
    public static void setup() {
        // Set the system property for the test
        System.setProperty("jnfe.base.service.provider.name", PROVIDER_NAME);
        System.setProperty("jnfe.base.service.provider.class.name", PROVIDER_CLASS_NAME);
    }

    @Test
    public void testAfterPropertiesSet() {
        try {
            bean.afterPropertiesSet();
        } catch (Exception e) {
            fail("Expected exception not thrown");
        }
        // Verify provider name
        assertEquals(PROVIDER_NAME, bean.getProviderName());
        // Verify provider class name
        assertEquals(PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }

    @Test
    public void testAfterPropertiesSet_defaultProvider() {
        try {
            bean.afterPropertiesSet();
        } catch (Exception e) {
            fail("Expected exception not thrown");
        }
        // Verify provider name
        assertEquals(DEFAULT_PROVIDER_NAME, bean.getProviderName());
        // Verify provider class name
        assertEquals(DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }

    @Test
    public void testAfterPropertiesSet_emptyProviderName() {
        bean.setProviderName(null);
        try {
            bean.afterPropertiesSet();
            fail("Expected exception thrown");
        } catch (Exception e) {
            // expected
        }
        // Verify provider name
        assertEquals(DEFAULT_PROVIDER_NAME, bean.getProviderName());
        // Verify provider class name
        assertEquals(DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }

    @Test
    public void testAfterPropertiesSet_emptyProviderClassName() {
        bean.setProviderClassName(null);
        try {
            bean.afterPropertiesSet();
            fail("Expected exception thrown");
        } catch (Exception e) {
            // expected
        }
        // Verify provider name
        assertEquals(DEFAULT_PROVIDER_NAME, bean.getProviderName());
        // Verify provider class name
        assertEquals(DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }

    @Test
    public void testAfterPropertiesSet_nullProviderName() {
        bean.setProviderName(null);
        try {
            bean.afterPropertiesSet();
            fail("Expected exception thrown");
        } catch (Exception e) {
            // expected
        }
        // Verify provider name
        assertEquals(DEFAULT_PROVIDER_NAME, bean.getProviderName());
        // Verify provider class name
        assertEquals(DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }

    @Test
    public void testAfterPropertiesSet_nullProviderClassName() {
        bean.setProviderClassName(null);
        try {
            bean.afterPropertiesSet();
            fail("Expected exception thrown");
        } catch (Exception e) {
            // expected
        }
        // Verify provider name
        assertEquals(DEFAULT_PROVIDER_NAME, bean.getProviderName());
        // Verify provider class name
        assertEquals(DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }
}
