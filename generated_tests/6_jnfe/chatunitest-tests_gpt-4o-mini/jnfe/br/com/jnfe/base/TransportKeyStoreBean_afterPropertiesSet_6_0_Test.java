package br.com.jnfe.base;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

public class TransportKeyStoreBean_afterPropertiesSet_6_0_Test {

    private TransportKeyStoreBean transportKeyStoreBean;

    private static final Logger logger = LoggerFactory.getLogger(TransportKeyStoreBean.class);

    @BeforeEach
    public void setUp() {
        transportKeyStoreBean = new TransportKeyStoreBean();
    }

    @Test
    public void testAfterPropertiesSet_WithNullKeyStoreTypeAndUri() throws Exception {
        // Arrange
        transportKeyStoreBean.setKeyStoreType(null);
        transportKeyStoreBean.setKeyStoreUri(null);
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        assertEquals("pkcs12", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("file:#{ systemProperties['user.home'] }/jnfe.pfx", System.getProperty("javax.net.ssl.keyStore"));
    }

    @Test
    public void testAfterPropertiesSet_WithEmptyKeyStoreTypeAndUri() throws Exception {
        // Arrange
        transportKeyStoreBean.setKeyStoreType("");
        transportKeyStoreBean.setKeyStoreUri("");
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        assertEquals("pkcs12", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("file:#{ systemProperties['user.home'] }/jnfe.pfx", System.getProperty("javax.net.ssl.keyStore"));
    }

    @Test
    public void testAfterPropertiesSet_WithValidKeyStoreTypeAndUri() throws Exception {
        // Arrange
        transportKeyStoreBean.setKeyStoreType("jks");
        transportKeyStoreBean.setKeyStoreUri("file:/path/to/keystore.jks");
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        assertEquals("jks", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("file:/path/to/keystore.jks", System.getProperty("javax.net.ssl.keyStore"));
    }

    @Test
    public void testAfterPropertiesSet_WithKeyStorePassword() throws Exception {
        // Arrange
        transportKeyStoreBean.setKeyStorePassword("password");
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        assertEquals("password", System.getProperty("javax.net.ssl.keyStorePassword"));
    }

    @Test
    public void testAfterPropertiesSet_WithTrustStoreProperties() throws Exception {
        // Arrange
        transportKeyStoreBean.setTrustStore("file:/path/to/truststore.jks");
        transportKeyStoreBean.setTrustStoreType("jks");
        transportKeyStoreBean.setTrustStorePassword("trustpassword");
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        assertEquals("file:/path/to/truststore.jks", System.getProperty("javax.net.ssl.trustStore"));
        assertEquals("jks", System.getProperty("javax.net.ssl.trustStoreType"));
        assertEquals("trustpassword", System.getProperty("javax.net.ssl.trustStorePassword"));
    }
}
