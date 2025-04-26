package br.com.jnfe.base;

import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

public class TransportKeyStoreBean_afterPropertiesSet_6_3_Test {

    @Test
    public void testAfterPropertiesSet() throws Exception {
        // Arrange
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        Mockito.doNothing().when(transportKeyStoreBean).setKeyStoreUri(Mockito.anyString());
        Mockito.doNothing().when(transportKeyStoreBean).setKeyStoreType(Mockito.anyString());
        Mockito.doNothing().when(transportKeyStoreBean).setKeyStorePassword(Mockito.anyString());
        Mockito.doNothing().when(transportKeyStoreBean).setTrustStore(Mockito.anyString());
        Mockito.doNothing().when(transportKeyStoreBean).setTrustStoreType(Mockito.anyString());
        Mockito.doNothing().when(transportKeyStoreBean).setTrustStorePassword(Mockito.anyString());
        // Act
        transportKeyStoreBean.afterPropertiesSet();
        // Assert
        Mockito.verify(transportKeyStoreBean).setKeyStoreUri(Mockito.anyString());
        Mockito.verify(transportKeyStoreBean).setKeyStoreType(Mockito.anyString());
        Mockito.verify(transportKeyStoreBean).setKeyStorePassword(Mockito.anyString());
        Mockito.verify(transportKeyStoreBean).setTrustStore(Mockito.anyString());
        Mockito.verify(transportKeyStoreBean).setTrustStoreType(Mockito.anyString());
        Mockito.verify(transportKeyStoreBean).setTrustStorePassword(Mockito.anyString());
    }
}
