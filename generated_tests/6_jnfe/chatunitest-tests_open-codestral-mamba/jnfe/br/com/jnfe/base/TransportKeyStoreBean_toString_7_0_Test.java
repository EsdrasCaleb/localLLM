package br.com.jnfe.base;

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

public class TransportKeyStoreBean_toString_7_0_Test {

    @InjectMocks
    private TransportKeyStoreBean transportKeyStoreBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        // Arrange
        String expectedString = "TransportKeyStoreBean@" + Integer.toHexString(transportKeyStoreBean.hashCode()) + " [javax.net.ssl.keyStore='null' javax.net.ssl.keyStoreType='null' javax.net.ssl.trustStoreType='null' javax.net.ssl.trustStore='null' ]";
        // Act
        String actualString = transportKeyStoreBean.toString();
        // Assert
        assertEquals(expectedString, actualString);
    }
}
