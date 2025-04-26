// Test method
package br.com.jnfe.base;

import com.sun.xml.bind.marshaller.NamespacePrefixMapper;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class DefaultNamespacePrefixMapper_getPreferredPrefix_0_4_Test {

    @Test
    public void testGetPreferredPrefixWithRequiredPrefix() {
        // Arrange
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        // Act
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = "nf";
        boolean requirePrefix = true;
        // Assert
        assertEquals("nf", mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix));
    }
}
