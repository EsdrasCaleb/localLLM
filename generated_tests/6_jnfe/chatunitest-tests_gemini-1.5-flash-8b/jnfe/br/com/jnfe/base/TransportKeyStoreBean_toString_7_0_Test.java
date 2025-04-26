package br.com.jnfe.base;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

class TransportKeyStoreBean_toString_7_0_Test {

    private TransportKeyStoreBean bean;

    @BeforeEach
    void setUp() {
        bean = new TransportKeyStoreBean();
    }

    @Test
    void testToString_defaultValues() {
        // Mock system properties for predictable output
        String expectedKeyStore = "someKeyStore";
        String expectedKeyStoreType = "someKeyStoreType";
        String expectedTrustStore = "someTrustStore";
        String expectedTrustStoreType = "someTrustStoreType";
        try {
            Field keyStoreField = TransportKeyStoreBean.class.getDeclaredField("keyStoreUri");
            keyStoreField.setAccessible(true);
            keyStoreField.set(bean, expectedKeyStore);
            Field keyStoreTypeField = TransportKeyStoreBean.class.getDeclaredField("keyStoreType");
            keyStoreTypeField.setAccessible(true);
            keyStoreTypeField.set(bean, expectedKeyStoreType);
            Field trustStoreField = TransportKeyStoreBean.class.getDeclaredField("trustStore");
            trustStoreField.setAccessible(true);
            trustStoreField.set(bean, expectedTrustStore);
            Field trustStoreTypeField = TransportKeyStoreBean.class.getDeclaredField("trustStoreType");
            trustStoreTypeField.setAccessible(true);
            trustStoreTypeField.set(bean, expectedTrustStoreType);
            // Mock system properties. Crucial for predictable output.
            System.setProperty("javax.net.ssl.keyStore", expectedKeyStore);
            System.setProperty("javax.net.ssl.keyStoreType", expectedKeyStoreType);
            System.setProperty("javax.net.ssl.trustStore", expectedTrustStore);
            System.setProperty("javax.net.ssl.trustStoreType", expectedTrustStoreType);
            String actual = bean.toString();
            assertTrue(actual.contains("javax.net.ssl.keyStore='" + expectedKeyStore + "'"));
            assertTrue(actual.contains("javax.net.ssl.keyStoreType='" + expectedKeyStoreType + "'"));
            assertTrue(actual.contains("javax.net.ssl.trustStore='" + expectedTrustStore + "'"));
            assertTrue(actual.contains("javax.net.ssl.trustStoreType='" + expectedTrustStoreType + "'"));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing fields: " + e.getMessage());
        }
    }
}
