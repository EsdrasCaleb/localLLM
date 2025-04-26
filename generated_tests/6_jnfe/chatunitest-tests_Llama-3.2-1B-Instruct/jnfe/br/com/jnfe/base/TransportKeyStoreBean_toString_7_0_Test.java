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

    @Test
    public void testToString() {
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        transportKeyStoreBean.setKeyStoreUri("file:#{user.home}/.ssh/id_rsa");
        transportKeyStoreBean.setKeyStoreType("pkcs12");
        transportKeyStoreBean.setKeyStorePassword("mykey");
        transportKeyStoreBean.setTrustStore("file:#{user.home}/.ssh/truststore");
        transportKeyStoreBean.setTrustStoreType("JKS");
        transportKeyStoreBean.setTrustStorePassword("mytrust");
        assertEquals("javax.net.ssl.keyStore='file:/Users/user/.ssh/id_rsa' " + "javax.net.ssl.keyStoreType='pkcs12' " + "javax.net.ssl.trustStoreType='JKS' " + "javax.net.ssl.trustStore='file:/Users/user/.ssh/truststore'", transportKeyStoreBean.toString());
    }
}
