package br.com.jnfe.base;

import javax.net.ssl.KeyManagerFactory;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableKeyException;
import java.security.cert.CertificateException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class TransportKeyStoreBean_openTransportKeyManagerFactory_9_0_Test {

    private TransportKeyStoreBean transportKeyStoreBean;

    @Mock
    private KeyStore mockKeyStore;

    @BeforeEach
    public void setup() {
        transportKeyStoreBean = new TransportKeyStoreBean();
    }

    @Test
    public void openTransportKeyManagerFactory_nullPassword() throws Exception {
        // Arrange
        String keyStorePassword = null;
        transportKeyStoreBean.setKeyStorePassword(keyStorePassword);
        // Act & Assert
        assertThrows(NullPointerException.class, () -> transportKeyStoreBean.openTransportKeyManagerFactory());
    }
}
