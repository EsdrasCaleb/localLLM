package br.com.jnfe.base.service;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableEntryException;
import java.security.cert.CertificateException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_loadKeyStore_4_0_Test {

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @Mock
    private Logger logger;

    @Mock
    private KeyStore keyStore;

    @Mock
    private InputStream inputStream;

    @BeforeEach
    public void setUp() {
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
    }

    @Test
    public void testLoadKeyStore_Success() throws Exception {
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenReturn(inputStream);
        when(keyStore.isKeyEntry("testAlias")).thenReturn(true);
        pkcs12SecurityHandlerBean.loadKeyStore();
        verify(logger).info("Aberto armazém {} localizado em {}.", keyStore, pkcs12SecurityHandlerBean.getLocation());
    }

    @Test
    public void testLoadKeyStore_FileNotFound() throws Exception {
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenThrow(new FileNotFoundException());
        assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
        verify(logger).warn("Armazém não localizado em {}.", pkcs12SecurityHandlerBean.getLocation());
    }

    @Test
    public void testLoadKeyStore_KeyEntryNotFound() throws Exception {
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenReturn(inputStream);
        when(keyStore.isKeyEntry("testAlias")).thenReturn(false);
        assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
        verify(logger).warn("Não existe chave particular para o alias '{}' em {}.", "testAlias", pkcs12SecurityHandlerBean.getLocation());
    }

    @Test
    public void testLoadKeyStore_GeneralException() throws Exception {
        when(pkcs12SecurityHandlerBean.getLocation().getInputStream()).thenThrow(new CertificateException("Test Exception"));
        assertThrows(RuntimeException.class, () -> pkcs12SecurityHandlerBean.loadKeyStore());
        verify(logger).warn("Erro ao abrir armazém localizado em {}.", pkcs12SecurityHandlerBean.getLocation());
    }
}
