package br.com.jnfe.base.service;

import static org.mockito.ArgumentMatchers.anyString;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_3_Test {

    @Mock
    private DOMNFeSigantureFactoryBean factoryBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAfterPropertiesSetWithDefaultValues() throws Exception {
        // Correcting the instantiation of XMLSignatureFactory
        when(factoryBean.getObject()).thenReturn(XMLSignatureFactory.getInstance());
        // Calling the afterPropertiesSet method
        factoryBean.afterPropertiesSet();
        // Verifying the default provider name and class name
        assertEquals("jsr105Provider", factoryBean.getProviderName());
        assertEquals("org.jcp.xml.dsig.internal.dom.XMLDSigRI", factoryBean.getProviderClassName());
    }

    @Test
    public void testAfterPropertiesSetWithExplicitValues() throws Exception {
        // Setting explicit values for provider name and class name
        factoryBean.setProviderName("myCustomProvider");
        factoryBean.setProviderClassName("com.example.MyCustomProvider");
        // Calling the afterPropertiesSet method
        factoryBean.afterPropertiesSet();
        // Verifying the explicitly set provider name and class name
        assertEquals("myCustomProvider", factoryBean.getProviderName());
        assertEquals("com.example.MyCustomProvider", factoryBean.getProviderClassName());
    }
}
