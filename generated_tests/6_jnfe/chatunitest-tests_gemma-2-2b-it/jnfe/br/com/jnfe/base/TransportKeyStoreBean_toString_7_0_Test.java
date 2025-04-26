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

public class TransportKeyStoreBean_toString_7_0_Test {

    @Test
    void testToString() {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        assertEquals(bean.toString(), "[javax.net.ssl.keyStore='file:#{ systemProperties['user.home'] }/jnfe.pfx' javax.net.ssl.keyStoreType='pkcs12' javax.net.ssl.trustStoreType='pkcs12' javax.net.ssl.trustStore='file:#{ systemProperties['user.home'] }/jnfe.pfx' ]");
    }
}
