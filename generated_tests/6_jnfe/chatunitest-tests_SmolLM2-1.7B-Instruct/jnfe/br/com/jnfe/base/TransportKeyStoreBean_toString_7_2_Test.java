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

public class TransportKeyStoreBean_toString_7_2_Test {

    @Test
    public void testToString() {
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        transportKeyStoreBean.setKeyStoreUri("file:#{ systemProperties['user.home'] }/jnfe.pfx");
        transportKeyStoreBean.setKeyStoreType("pkcs12");
        transportKeyStoreBean.setKeyStorePassword("password");
        transportKeyStoreBean.setTrustStore("file:#{ systemProperties['user.home'] }/jnfe.jks");
        transportKeyStoreBean.setTrustStoreType("jks");
        transportKeyStoreBean.setTrustStorePassword("password");
        String expected = "TransportKeyStoreBean@" + Integer.toHexString(hashCode()) + " [javax.net.ssl.keyStore='file:#{ systemProperties['user.home'] }/jnfe.pfx' " + "javax.net.ssl.keyStoreType='pkcs12' " + "javax.net.ssl.trustStoreType='jks' " + "javax.net.ssl.trustStore='file:#{ systemProperties['user.home'] }/jnfe.jks' " + "]";
        assertNotNull(transportKeyStoreBean.toString());
        assertEquals(expected, transportKeyStoreBean.toString());
    }
}
