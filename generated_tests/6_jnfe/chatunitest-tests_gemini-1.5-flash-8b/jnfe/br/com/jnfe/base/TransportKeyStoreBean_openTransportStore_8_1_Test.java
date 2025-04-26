package br.com.jnfe.base;

import java.security.KeyStore;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.net.ssl.KeyManagerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

class TransportKeyStoreBean_openTransportStore_8_1_Test {

    private TransportKeyStoreBean transportKeyStoreBean;

    private SecurityUtils mockSecurityUtils;

    @BeforeEach
    void setUp() {
        transportKeyStoreBean = new TransportKeyStoreBean();
        mockSecurityUtils = Mockito.mock(SecurityUtils.class);
        try {
            // Crucial:  Prevent NullPointerExceptions if you don't set these values
            transportKeyStoreBean.setKeyStoreType("pkcs12");
            transportKeyStoreBean.setKeyStoreUri("testUri");
            transportKeyStoreBean.setKeyStorePassword("testPassword");
        } catch (Exception e) {
            fail("Unexpected exception during setup: " + e.getMessage());
        }
        // Crucially, set the SecurityUtils instance
        try {
            java.lang.reflect.Field securityUtilsField = TransportKeyStoreBean.class.getDeclaredField("logger");
            securityUtilsField.setAccessible(true);
            securityUtilsField.set(transportKeyStoreBean, Mockito.mock(java.util.logging.Logger.class));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set logger: " + e.getMessage());
        }
        try {
            java.lang.reflect.Field securityUtilsField = TransportKeyStoreBean.class.getDeclaredField("logger");
            securityUtilsField.setAccessible(true);
            securityUtilsField.set(transportKeyStoreBean, Mockito.mock(java.util.logging.Logger.class));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set logger: " + e.getMessage());
        }
    }

    @Test
    void openTransportStore_success() throws Exception {
        KeyStore mockKeyStore = Mockito.mock(KeyStore.class);
        when(mockSecurityUtils.openStore("pkcs12", "testUri", "testPassword".toCharArray())).thenReturn(mockKeyStore);
        KeyStore keyStore = transportKeyStoreBean.openTransportStore();
        assertNotNull(keyStore);
        verify(mockSecurityUtils).openStore("pkcs12", "testUri", "testPassword".toCharArray());
    }

    @Test
    void openTransportStore_failure() throws Exception {
        when(mockSecurityUtils.openStore("pkcs12", "testUri", "testPassword".toCharArray())).thenThrow(new IOException("Failed to open keystore"));
        assertThrows(IOException.class, () -> transportKeyStoreBean.openTransportStore());
    }
}
