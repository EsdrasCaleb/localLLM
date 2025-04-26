package br.com.jnfe.base;

import java.lang.reflect.Field;
import java.util.Properties;
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

public class TransportKeyStoreBean_toString_7_0_Test {

    @Test
    void testToString() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        // Test with default system properties (or unset properties)
        String toStringDefault = bean.toString();
        assertTrue(toStringDefault.contains("TransportKeyStoreBean@"));
        assertTrue(toStringDefault.contains("javax.net.ssl.keyStore='"));
        assertTrue(toStringDefault.contains("javax.net.ssl.keyStoreType='"));
        assertTrue(toStringDefault.contains("javax.net.ssl.trustStoreType='"));
        assertTrue(toStringDefault.contains("javax.net.ssl.trustStore='"));
        // Test with custom system properties
        Properties props = System.getProperties();
        props.setProperty("javax.net.ssl.keyStore", "myKeyStore");
        props.setProperty("javax.net.ssl.keyStoreType", "JKS");
        props.setProperty("javax.net.ssl.trustStoreType", "PKCS12");
        props.setProperty("javax.net.ssl.trustStore", "myTrustStore");
        System.setProperties(props);
        String toStringCustom = bean.toString();
        assertTrue(toStringCustom.contains("TransportKeyStoreBean@"));
        assertTrue(toStringCustom.contains("javax.net.ssl.keyStore='myKeyStore'"));
        assertTrue(toStringCustom.contains("javax.net.ssl.keyStoreType='JKS'"));
        assertTrue(toStringCustom.contains("javax.net.ssl.trustStoreType='PKCS12'"));
        assertTrue(toStringCustom.contains("javax.net.ssl.trustStore='myTrustStore'"));
        // Restore system properties
        System.setProperties(new Properties());
    }
}
