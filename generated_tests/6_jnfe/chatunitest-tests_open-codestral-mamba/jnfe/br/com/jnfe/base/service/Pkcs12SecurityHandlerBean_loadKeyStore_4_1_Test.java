package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.security.KeyStore;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableEntryException;
import java.security.cert.X509Certificate;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_loadKeyStore_4_1_Test {

    @Mock
    private Logger logger;

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testLoadKeyStore_Success() throws Exception {
        // Arrange
        String alias = "testAlias";
        char[] password = "testPassword".toCharArray();
        pkcs12SecurityHandlerBean.setAlias(alias);
        pkcs12SecurityHandlerBean.setPassword(password);
        KeyStore mockKeyStore = mock(KeyStore.class);
        when(mockKeyStore.isKeyEntry(alias)).thenReturn(true);
        InputStream mockInputStream = mock(FileInputStream.class);
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenReturn(mockInputStream);
        doNothing().when(pkcs12SecurityHandlerBean).setKeyStore(mockKeyStore);
        // Act
        pkcs12SecurityHandlerBean.loadKeyStore();
        // Assert
        // No assertion needed as the method is void
    }

    @Test
    public void testLoadKeyStore_FileNotFoundException() throws Exception {
        // Arrange
        String alias = "testAlias";
        char[] password = "testPassword".toCharArray();
        pkcs12SecurityHandlerBean.setAlias(alias);
        pkcs12SecurityHandlerBean.setPassword(password);
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenThrow(FileNotFoundException.class);
        // Act & Assert
        assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
    }

    @Test
    public void testLoadKeyStore_Exception() throws Exception {
        // Arrange
        String alias = "testAlias";
        char[] password = "testPassword".toCharArray();
        pkcs12SecurityHandlerBean.setAlias(alias);
        pkcs12SecurityHandlerBean.setPassword(password);
        KeyStore mockKeyStore = mock(KeyStore.class);
        when(mockKeyStore.isKeyEntry(alias)).thenReturn(true);
        InputStream mockInputStream = mock(FileInputStream.class);
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenReturn(mockInputStream);
        doThrow(new RuntimeException()).when(pkcs12SecurityHandlerBean).setKeyStore(mockKeyStore);
        // Act & Assert
        assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
    }
}
