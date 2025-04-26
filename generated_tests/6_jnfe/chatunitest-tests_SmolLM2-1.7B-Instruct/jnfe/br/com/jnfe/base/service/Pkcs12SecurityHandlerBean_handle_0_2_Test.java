package br.com.jnfe.base.service;

import org.junit.Test;
import static org.junit.Assert.*;
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

public class Pkcs12SecurityHandlerBean_handle_0_2_Test {

    @Test
    public void testHandle() {
        // Arrange
        Pkcs12SecurityHandlerBean handler = new Pkcs12SecurityHandlerBean();
        SecurityCallBack action = Mockito.mock(SecurityCallBack.class);
        // Act
        handler.handle(null, null, action);
        // Assert
        Mockito.verify(action, Mockito.times(1)).doInSecurityContext(null, null, null, null);
    }
}
