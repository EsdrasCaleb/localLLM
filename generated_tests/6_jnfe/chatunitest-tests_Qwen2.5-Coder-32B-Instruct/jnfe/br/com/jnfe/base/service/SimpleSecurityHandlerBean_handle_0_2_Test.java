package br.com.jnfe.base.service;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.Principal;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import java.util.Vector;
import javax.xml.crypto.dsig.keyinfo.KeyInfo;
import org.slf4j.Logger;
import org.w3c.dom.Element;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class SimpleSecurityHandlerBean_handle_0_2_Test {

    @Mock
    private Logger logger;

    @Mock
    private KeyStore keyStore;

    @Mock
    private PrivateKey privateKey;

    @Mock
    private Certificate certificate;

    @Mock
    private X509Certificate x509Certificate;

    @Mock
    private SecurityCallBack securityCallBack;

    @Mock
    private Element parentElement;

    @Mock
    private Element elementToSign;

    @Mock
    private Enumeration<String> aliases;

    @InjectMocks
    private SimpleSecurityHandlerBean simpleSecurityHandlerBean;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        when(keyStore.getType()).thenReturn("JKS");
        when(keyStore.containsAlias("testAlias")).thenReturn(true);
        when(keyStore.getKey("testAlias", "testPassword".toCharArray())).thenReturn(privateKey);
        when(keyStore.getCertificate("testAlias")).thenReturn(certificate);
        when(certificate instanceof X509Certificate).thenReturn(true);
        when(keyStore.aliases()).thenReturn(aliases);
        when(aliases.hasMoreElements()).thenReturn(true);
        when(aliases.nextElement()).thenReturn("defaultAlias");
        System.setProperty("javax.net.ssl.keyStore", "testKeyStorePath");
        System.setProperty("javax.net.ssl.keyStoreType", "JKS");
        System.setProperty("javax.net.ssl.keyStorePassword", "testPassword");
    }

    @Test
    public void testHandleWithConfiguredKeyStore() throws Exception {
        simpleSecurityHandlerBean.setAlias("testAlias");
        simpleSecurityHandlerBean.setPassword("testPassword".toCharArray());
        simpleSecurityHandlerBean.handle(parentElement, elementToSign, securityCallBack);
        verify(logger).debug("Recuperando credenciais de armazém tipo JKS.");
        verify(logger).debug("Chave particular recuperada no formato: {}", privateKey.getFormat());
        verify(logger).debug("Certificado recuperado: {}", x509Certificate.getSubjectDN());
        verify(securityCallBack).doInSecurityContext(parentElement, elementToSign, certificate, privateKey);
    }

    @Test
    public void testHandleWithConfiguredKeyStoreAliasNotFound() throws Exception {
        when(keyStore.containsAlias("testAlias")).thenReturn(false);
        simpleSecurityHandlerBean.setAlias("testAlias");
        simpleSecurityHandlerBean.setPassword("testPassword".toCharArray());
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            simpleSecurityHandlerBean.handle(parentElement, elementToSign, securityCallBack);
        });
        String expectedMessage = "Armazém configurado pelo bean 'keyStore' não contém o certificado 'testAlias'. Tente outro 'alias' ou reconfigure jnfe-core-context.xml para evitar a criação do bean 'keyStore', forçando o sistema a usar o armazém principal.";
        assertEquals(expectedMessage, exception.getMessage());
    }

    @Test
    public void testHandleWithDefaultKeyStore() throws Exception {
        when(keyStore.containsAlias("testAlias")).thenReturn(false);
        when(keyStore.containsAlias("defaultAlias")).thenReturn(true);
        when(keyStore.getKey("defaultAlias", "testPassword".toCharArray())).thenReturn(privateKey);
        when(keyStore.getCertificate("defaultAlias")).thenReturn(certificate);
        when(certificate instanceof X509Certificate).thenReturn(true);
        when(((X509Certificate) certificate).getSubjectDN()).thenReturn(mock(Principal.class));
        simpleSecurityHandlerBean.setAlias("testAlias");
        simpleSecurityHandlerBean.setPassword("testPassword".toCharArray());
        simpleSecurityHandlerBean.handle(parentElement, elementToSign, securityCallBack);
        verify(logger).debug("Recuperando credenciais da primeira chave do armazém principal em testKeyStorePath.");
        verify(logger).debug("Certificado: {}", ((X509Certificate) certificate).getSubjectDN());
        verify(securityCallBack).doInSecurityContext(parentElement, elementToSign, certificate, privateKey);
    }

    @Test
    public void testHandleWithDefaultKeyStoreNoAliases() throws Exception {
        when(keyStore.containsAlias("testAlias")).thenReturn(false);
    }
}
