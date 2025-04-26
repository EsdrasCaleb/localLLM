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

class TransportKeyStoreBean_afterPropertiesSet_6_0_Test {

    @Test
    void afterPropertiesSet_defaultValues_setsProperties() throws Exception {
        // Arrange
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        Field[] fields = transportKeyStoreBean.getClass().getDeclaredFields();
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        for (Field field : fields) {
            field.setAccessible(true);
            Object value = field.get(transportKeyStoreBean);
            assertNotNull(value);
        }
    }

    @Test
    void afterPropertiesSet_customValues_setsProperties() throws Exception {
        // Arrange
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        String keyStoreUri = "customUri";
        String keyStoreType = "customType";
        String keyStorePassword = "customPassword";
        String trustStore = "customTrustStore";
        String trustStoreType = "customTrustStoreType";
        String trustStorePassword = "customTrustStorePassword";
        // Act
        transportKeyStoreBean.setKeyStoreUri(keyStoreUri);
        transportKeyStoreBean.setKeyStoreType(keyStoreType);
        transportKeyStoreBean.setKeyStorePassword(keyStorePassword);
        transportKeyStoreBean.setTrustStore(trustStore);
        transportKeyStoreBean.setTrustStoreType(trustStoreType);
        transportKeyStoreBean.setTrustStorePassword(trustStorePassword);
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        assertEquals(keyStoreUri, System.getProperty("javax.net.ssl.keyStore"));
        assertEquals(keyStoreType, System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals(keyStorePassword, System.getProperty("javax.net.ssl.keyStorePassword"));
        assertEquals(trustStore, System.getProperty("javax.net.ssl.trustStore"));
        assertEquals(trustStoreType, System.getProperty("javax.net.ssl.trustStoreType"));
        assertEquals(trustStorePassword, System.getProperty("javax.net.ssl.trustStorePassword"));
    }
}
