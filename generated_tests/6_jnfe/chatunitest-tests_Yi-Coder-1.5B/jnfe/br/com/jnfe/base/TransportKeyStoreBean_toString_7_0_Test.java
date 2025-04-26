package br.com.jnfe.base;

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

@ExtendWith(MockitoExtension.class)
public class TransportKeyStoreBean_toString_7_0_Test {

    // Test class
    @Test
    public void testToString() {
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        transportKeyStoreBean.setKeyStoreUri("file:C:\\Users\\user\\jnfe.pfx");
        transportKeyStoreBean.setKeyStoreType("pkcs12");
        transportKeyStoreBean.setKeyStorePassword("123456");
        transportKeyStoreBean.setTrustStore("file:C:\\Users\\user\\jnfe.pfx");
        transportKeyStoreBean.setTrustStoreType("pkcs12");
        transportKeyStoreBean.setTrustStorePassword("123456");
        String expected = "TransportKeyStoreBean@16f98625 [javax.net.ssl.keyStore='file:C:\\Users\\user\\jnfe.pfx' javax.net.ssl.keyStoreType='pkcs12' javax.net.ssl.trustStoreType='pkcs12' javax.net.ssl.trustStore='file:C:\\Users\\user\\jnfe.pfx']";
        assertEquals(expected, transportKeyStoreBean.toString());
    }
}
