package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;

@ExtendWith(MockitoExtension.class)
public class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_0_Test {

    @InjectMocks
    private DOMNFeSigantureFactoryBean domNFeSigantureFactoryBean;

    @Test
    public void afterPropertiesSetTest() throws Exception {
        String defaultProviderName = DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME;
        String defaultProviderClassName = DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME;
        // test if providerName is set to default
        domNFeSigantureFactoryBean.afterPropertiesSet();
        assertEquals(defaultProviderName, domNFeSigantureFactoryBean.getProviderName());
        // test if providerClassName is set to default
        domNFeSigantureFactoryBean.afterPropertiesSet();
        assertEquals(defaultProviderClassName, domNFeSigantureFactoryBean.getProviderClassName());
        // test if providerName is set to system property
        String customProviderName = "customProviderName";
        String customProviderClassName = "customProviderClassName";
        System.setProperty("providerName", customProviderName);
        System.setProperty("providerClassName", customProviderClassName);
        domNFeSigantureFactoryBean.afterPropertiesSet();
        assertEquals(customProviderName, domNFeSigantureFactoryBean.getProviderName());
        assertEquals(customProviderClassName, domNFeSigantureFactoryBean.getProviderClassName());
    }
}
