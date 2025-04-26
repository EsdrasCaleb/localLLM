package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.w3c.dom.Element;
import javax.security.auth.x500.X500Principal;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import static org.mockito.ArgumentMatchers.any;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@ExtendWith(MockitoExtension.class)
public class SimpleSecurityHandlerBean_handle_0_0_Test {

    @Mock
    private KeyStore keyStore;

    @Mock
    private SecurityCallBack action;

    @Mock
    private Element parentElement;

    @Mock
    private Element elementToSign;

    @InjectMocks
    private SimpleSecurityHandlerBean simpleSecurityHandlerBean;

    @Test
    void testHandle_CustomKeyStore_Success() throws Exception {
        // Mocking necessary objects and behaviors
        String alias = "testAlias";
        char[] password = "password".toCharArray();
        PrivateKey privateKey = mock(PrivateKey.class);
        X509Certificate certificate = mock(X509Certificate.class);
        when(keyStore.getType()).thenReturn("JKS");
        when(keyStore.containsAlias(alias)).thenReturn(true);
        when(keyStore.getKey(alias, password)).thenReturn(privateKey);
        when(keyStore.getCertificate(alias)).thenReturn(certificate);
        when(certificate.getSubjectDN()).thenReturn(new X500Principal("CN=Test"));
        simpleSecurityHandlerBean.setAlias(alias);
        simpleSecurityHandlerBean.setPassword(password);
        when(simpleSecurityHandlerBean.getKeyStore()).thenReturn(keyStore);
        // Invoke the method
        simpleSecurityHandlerBean.handle(parentElement, elementToSign, action);
        // Verify interactions
        verify(keyStore, times(1)).containsAlias(alias);
        verify(keyStore, times(1)).getKey(alias, password);
        verify(keyStore, times(1)).getCertificate(alias);
        verify(action, times(1)).doInSecurityContext(parentElement, elementToSign, certificate, privateKey);
    }

    @Test
    void testHandle_CustomKeyStore_Failure() throws Exception {
        // Mocking necessary objects and behaviors
        String alias = "testAlias";
        char[] password = "password".toCharArray();
        simpleSecurityHandlerBean.setAlias(alias);
        simpleSecurityHandlerBean.setPassword(password);
        when(simpleSecurityHandlerBean.getKeyStore()).thenReturn(keyStore);
        when(keyStore.containsAlias(alias)).thenReturn(false);
        // Invoke the method, expect exception
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            simpleSecurityHandlerBean.handle(parentElement, elementToSign, action);
        });
        // Verify exception message
        assertTrue(exception.getMessage().contains("n�o cont�m o certificado"));
        verify(keyStore, times(1)).containsAlias(alias);
        verify(keyStore, never()).getKey(anyString(), any(char[].class));
        verify(keyStore, never()).getCertificate(anyString());
        verify(action, never()).doInSecurityContext(any(), any(), any(), any());
    }

    @Test
    void testHandle_DefaultKeyStore_Success() throws Exception {
        // Mocking necessary objects and behaviors
        String alias = "testAlias";
        char[] password = "password".toCharArray();
        PrivateKey privateKey = mock(PrivateKey.class);
        X509Certificate certificate = mock(X509Certificate.class);
        KeyStore ksKeys = mock(KeyStore.class);
        Enumeration<String> aliases = mock(Enumeration.class);
        when(aliases.hasMoreElements()).thenReturn(true, false);
        when(aliases.nextElement()).thenReturn(alias);
        when(ksKeys.aliases()).thenReturn(aliases);
        when(ksKeys.getCertificate(alias)).thenReturn(certificate);
        when(ksKeys.getKey(alias, password)).thenReturn(privateKey);
        when(certificate.getSubjectDN()).thenReturn(new X500Principal("CN=Test"));
        when(simpleSecurityHandlerBean.getKeyStore()).thenReturn(null);
    }
}
