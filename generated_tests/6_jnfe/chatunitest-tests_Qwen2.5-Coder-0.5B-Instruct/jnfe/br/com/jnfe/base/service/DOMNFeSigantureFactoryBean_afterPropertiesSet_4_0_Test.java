package br.com.jnfe.base.service;

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

class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_0_Test {

    private DOMNFeSigantureFactoryBean factoryBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        this.factoryBean = new DOMNFeSigantureFactoryBean();
    }

    @Test
    public void testAfterPropertiesSet() {
        // Setup the provider name and provider class name
        factoryBean.setProviderName("org.jcp.xml.dsig.internal.dom.XMLDSigRI");
        factoryBean.setProviderClassName("jsr105Provider");
        // <Buggy Line>: unreported exception java.lang.Exception; must be caught or declared to be thrown
        try {
            factoryBean.afterPropertiesSet();
        } catch (Exception e) {
            fail("Expected an exception, but got " + e.getMessage());
        }
        // Verify that the provider name is set
        assertEquals("org.jcp.xml.dsig.internal.dom.XMLDSigRI", factoryBean.getProviderName());
        // Verify that the provider class name is set
        assertEquals("jsr105Provider", factoryBean.getProviderClassName());
    }
}
