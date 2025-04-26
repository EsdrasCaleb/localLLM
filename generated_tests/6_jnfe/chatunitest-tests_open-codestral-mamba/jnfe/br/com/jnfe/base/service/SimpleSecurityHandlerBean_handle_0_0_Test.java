package br.com.jnfe.base.service;

import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileInputStream;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class SimpleSecurityHandlerBean_handle_0_0_Test {

    @Mock
    private KeyStore keyStore;

    @Mock
    private Element parentElement;

    @Mock
    private Element elementToSign;

    @Mock
    private SecurityCallBack action;

    @InjectMocks
    private SimpleSecurityHandlerBean handler;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testHandle() throws Exception {
        PrivateKey privateKey = mock(PrivateKey.class);
        Certificate certificate = mock(Certificate.class);
        when(keyStore.containsAlias(anyString())).thenReturn(true);
        when(keyStore.getKey(anyString(), any(char[].class))).thenReturn(privateKey);
        when(keyStore.getCertificate(anyString())).thenReturn(certificate);
        handler.setKeyStore(keyStore);
        handler.handle(parentElement, elementToSign, action);
        verify(action).doInSecurityContext(parentElement, elementToSign, certificate, privateKey);
    }
}
