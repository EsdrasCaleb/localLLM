package br.com.jnfe.base;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sun.xml.bind.marshaller.NamespacePrefixMapper;

class DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test {

    @ParameterizedTest
    @CsvSource({ "http://www.portalfiscal.inf.br/nfe,nf,true", "http://www.portalfiscal.inf.br/nfe,suggest,true", "http://www.portalfiscal.inf.br/nfe,suggest,false", "http://www.w3.org/2000/09/xmldsig#,null,true", "http://www.w3.org/2000/09/xmldsig#,suggest,false", "http://www.example.com,null,true", "http://www.example.com,suggest,false" })
    void getPreferredPrefix_shouldReturnCorrectPrefix(String namespaceUri, String suggestion, boolean requirePrefix) {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String result = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        if (namespaceUri.equals("http://www.portalfiscal.inf.br/nfe")) {
            assertEquals("nf", result);
        } else {
            assertEquals(null, result);
        }
    }
}
