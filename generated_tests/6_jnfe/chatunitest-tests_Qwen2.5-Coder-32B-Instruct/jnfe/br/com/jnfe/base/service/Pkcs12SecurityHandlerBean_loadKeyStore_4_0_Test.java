package br.com.jnfe.base.service;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableEntryException;
import java.security.cert.X509Certificate;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.PrivateKey;
import org.w3c.dom.Element;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_loadKeyStore_4_0_Test {

    @Mock
    private KeyStore mockKeyStore;

    @Mock
    private FileInputStream mockFileInputStream;

    @Spy
    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    private static final Logger logger = LoggerFactory.getLogger(Pkcs12SecurityHandlerBean.class);

    @BeforeEach
    public void setUp() throws NoSuchAlgorithmException, KeyStoreException {
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        doReturn(mockKeyStore).when(pkcs12SecurityHandlerBean).getKeyStore();
        try {
            doReturn(mockFileInputStream).when(pkcs12SecurityHandlerBean).getLocation().getInputStream();
        } catch (IOException e) {
            fail("Unexpected IOException during setup", e);
        }
    }

    @Test
    public void testLoadKeyStore_Success() throws Exception {
        when(mockKeyStore.isKeyEntry("testAlias")).thenReturn(true);
        doNothing().when(pkcs12SecurityHandlerBean).setKeyStore(mockKeyStore);
        assertDoesNotThrow(() -> pkcs12SecurityHandlerBean.loadKeyStore());
        verify(mockKeyStore).load(mockFileInputStream, "testPassword".toCharArray());
        verify(pkcs12SecurityHandlerBean).setKeyStore(mockKeyStore);
        verify(logger).info("Aberto armazém {} localizado em {}.", mockKeyStore, pkcs12SecurityHandlerBean.getLocation());
    }

    @Test
    public void testLoadKeyStore_NoKeyEntry() throws Exception {
        when(mockKeyStore.isKeyEntry("testAlias")).thenReturn(false);
        doNothing().when(pkcs12SecurityHandlerBean).setKeyStore(mockKeyStore);
        Exception exception = assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
        assertEquals("Não existe chave particular no armazém designado.", exception.getMessage());
        verify(mockKeyStore).load(mockFileInputStream, "testPassword".toCharArray());
        verify(pkcs12SecurityHandlerBean).setKeyStore(mockKeyStore);
        verify(logger).warn("Não existe chave particular para o alias '{}' em {}.", "testAlias", pkcs12SecurityHandlerBean.getLocation());
    }

    @Test
    public void testLoadKeyStore_FileNotFoundException() throws Exception {
        doThrow(new FileNotFoundException()).when(pkcs12SecurityHandlerBean).getLocation().getInputStream();
        Exception exception = assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
        assertEquals("Erro ao abrir armazém, ", exception.getMessage());
        verify(logger).warn("Armazém não localizado em {}.", pkcs12SecurityHandlerBean.getLocation());
    }

    @Test
    public void testLoadKeyStore_Exception() throws Exception {
        doThrow(new IOException()).when(mockKeyStore).load(mockFileInputStream, "testPassword".toCharArray());
        Exception exception = assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
        assertEquals("Erro ao abrir armazém, ", exception.getMessage());
        verify(logger).warn("Erro ao abrir armazém localizado em {}.", pkcs12SecurityHandlerBean.getLocation());
    }
}
