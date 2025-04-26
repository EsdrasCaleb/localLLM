package br.com.jnfe.base.service;

// Test class
import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileNotFoundException;
import java.security.KeyStore;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableEntryException;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

@RunWith(MockitoJUnitRunner.class)
public class Pkcs12SecurityHandlerBean_handle_0_0_Test {

    @Mock
    private Element sourceElement;

    @Mock
    private Element elementToSign;

    @Mock
    private SecurityCallBack action;

    @Test
    public void testHandle() {
        Pkcs12SecurityHandlerBean sut = new Pkcs12SecurityHandlerBean();
        sut.setAlias("alias");
        sut.setPassword("password".toCharArray());
        sut.handle(sourceElement, elementToSign, action);
        Mockito.verify(action, Mockito.times(1)).doInSecurityContext(sourceElement, elementToSign, Mockito.any(), Mockito.any());
    }
}
