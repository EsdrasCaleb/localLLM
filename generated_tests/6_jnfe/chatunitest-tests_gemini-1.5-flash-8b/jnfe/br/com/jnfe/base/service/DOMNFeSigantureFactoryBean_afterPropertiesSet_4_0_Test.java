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
class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_0_Test {

    @Mock
    private org.slf4j.Logger logger;

    @InjectMocks
    private DOMNFeSigantureFactoryBean factoryBean;

    @Test
    void testAfterPropertiesSet_providerNameEmpty() throws Exception {
        // Set providerClassName to empty string
        factoryBean.setProviderClassName("");
        // Invoke the method under test
        factoryBean.afterPropertiesSet();
        // Verify that the providerName was set to the default
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, factoryBean.getProviderName());
        // Verify the logger warning message was logged
        verify(logger).warn("PRovider name n�o definido, usando {}", DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME);
    }

    @Test
    void testAfterPropertiesSet_providerNameAndClassNameEmpty() throws Exception {
        // Set both providerClassName and providerName to empty strings
        factoryBean.setProviderClassName("");
        factoryBean.setProviderName("");
        // Invoke the method under test
        factoryBean.afterPropertiesSet();
        // Verify that the providerName was set to the default
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, factoryBean.getProviderName());
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, factoryBean.getProviderClassName());
        // Verify the logger warning messages were logged
        verify(logger).warn("PRovider name n�o definido, usando {}", DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME);
        verify(logger).warn("PRovider name n�o definido, usando {}", DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME);
    }

    @Test
    void testAfterPropertiesSet_providerNameAndClassNameNotEmpty() throws Exception {
        String providerName = "testProvider";
        String providerClassName = "testClassName";
        factoryBean.setProviderName(providerName);
        factoryBean.setProviderClassName(providerClassName);
        factoryBean.afterPropertiesSet();
        assertEquals(providerName, factoryBean.getProviderName());
        // Verify no warnings were logged
        verify(logger, never()).warn(Mockito.anyString());
    }
}
