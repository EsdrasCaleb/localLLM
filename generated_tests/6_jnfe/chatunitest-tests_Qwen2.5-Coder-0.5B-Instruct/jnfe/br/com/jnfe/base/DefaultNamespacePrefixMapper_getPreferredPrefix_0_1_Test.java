package br.com.jnfe.base;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sun.xml.bind.marshaller.NamespacePrefixMapper;

class DefaultNamespacePrefixMapper_getPreferredPrefix_0_1_Test {

    @Test
    void testGetPreferredPrefix() {
        // Arrange
        DefaultNamespacePrefixMapper mapper = Mockito.mock(DefaultNamespacePrefixMapper.class);
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = "http://www.w3.org/2000/09/xmldsig#";
        boolean requirePrefix = true;
        // Act
        String preferredPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        // Assert
        Mockito.verify(mapper).getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals("nf", preferredPrefix);
    }
}
