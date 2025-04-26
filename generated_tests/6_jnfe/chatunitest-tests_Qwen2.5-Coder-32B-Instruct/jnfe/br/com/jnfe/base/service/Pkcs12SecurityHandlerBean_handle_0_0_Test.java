package br.com.jnfe.base.service;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import org.w3c.dom.Element;
import javax.xml.crypto.dsig.keyinfo.KeyInfo;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileNotFoundException;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableEntryException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_handle_0_0_Test {

    @Mock
    private Element sourceElement;

    @Mock
    private Element elementToSign;

    @Mock
    private SecurityCallBack securityCallBack;

    @Mock
    private KeyStore keyStore;

    @Mock
    private KeyStore.PrivateKeyEntry privateKeyEntry;

    @Mock
    private PrivateKey privateKey;

    @Mock
    private X509Certificate x509Certificate;

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
    }

    @Test
    public void testHandle_Success() throws Exception {
        when(privateKeyEntry.getPrivateKey()).thenReturn(privateKey);
        when(privateKeyEntry.getCertificate()).thenReturn(x509Certificate);
        when(unlockPkEntry(pkcs12SecurityHandlerBean)).thenReturn(privateKeyEntry);
        pkcs12SecurityHandlerBean.handle(sourceElement, elementToSign, securityCallBack);
        verify(securityCallBack, times(1)).doInSecurityContext(eq(sourceElement), eq(elementToSign), eq(x509Certificate), eq(privateKey));
    }

    @Test
    public void testHandle_Exception() throws Exception {
        when(unlockPkEntry(pkcs12SecurityHandlerBean)).thenThrow(new NoSuchAlgorithmException());
        IllegalArgumentException exception = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            pkcs12SecurityHandlerBean.handle(sourceElement, elementToSign, securityCallBack);
        });
        Assertions.assertEquals("Impossível recuperar credenciais", exception.getMessage());
        verify(securityCallBack, never()).doInSecurityContext(any(), any(), any(), any());
    }

    private PrivateKeyEntry unlockPkEntry(Pkcs12SecurityHandlerBean bean) throws Exception {
        Method method = Pkcs12SecurityHandlerBean.class.getDeclaredMethod("unlockPkEntry");
        method.setAccessible(true);
        return (PrivateKeyEntry) method.invoke(bean);
    }
}
