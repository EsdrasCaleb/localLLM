package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.w3c.dom.Element;
import java.security.cert.X509Certificate;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileNotFoundException;
import java.security.KeyStore;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableEntryException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_handle_0_0_Test {

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @Mock
    private SecurityCallBack action;

    @Mock
    private KeyStore.PrivateKeyEntry pkEntry;

    @Mock
    private PrivateKey privateKey;

    @Mock
    private X509Certificate certificate;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testHandle() {
        Element sourceElement = mock(Element.class);
        Element elementToSign = mock(Element.class);
        when(pkEntry.getPrivateKey()).thenReturn(privateKey);
        when(pkEntry.getCertificate()).thenReturn(certificate);
        pkcs12SecurityHandlerBean.handle(sourceElement, elementToSign, action);
        verify(action).doInSecurityContext(sourceElement, elementToSign, certificate, privateKey);
    }

    @Test
    public void testHandleException() {
        Element sourceElement = mock(Element.class);
        Element elementToSign = mock(Element.class);
        when(pkEntry.getPrivateKey()).thenThrow(new RuntimeException("Failed to get private key"));
        assertThrows(IllegalArgumentException.class, () -> pkcs12SecurityHandlerBean.handle(sourceElement, elementToSign, action));
    }
}
