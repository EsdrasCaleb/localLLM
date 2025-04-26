package br.com.jnfe.base.service;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;

public class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_0_Test {

    private DOMNFeSigantureFactoryBean bean;

    @BeforeEach
    public void setUp() {
        bean = new DOMNFeSigantureFactoryBean();
    }

    @Test
    public void testAfterPropertiesSet() throws Exception {
        bean.setProviderName("");
        bean.setProviderClassName("");
        bean.afterPropertiesSet();
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, bean.getProviderName());
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }
}
