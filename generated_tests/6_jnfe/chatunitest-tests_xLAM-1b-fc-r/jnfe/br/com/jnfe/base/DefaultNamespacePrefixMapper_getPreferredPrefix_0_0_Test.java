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

public class DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test {

    @Test
    public void getPreferredPrefix_returnsCorrectPrefix_whenNamespaceUriIsCorrect() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = "mySuggestion";
        boolean requirePrefix = true;
        String expectedPrefix = "nf";
        String actualPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals(expectedPrefix, actualPrefix);
    }

    @Test
    public void getPreferredPrefix_returnsNull_whenNamespaceUriIsNotCorrect() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String namespaceUri = "http://www.w3.org/2000/09/xmldsig#";
        String suggestion = "mySuggestion";
        boolean requirePrefix = true;
        String actualPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals(null, actualPrefix);
    }
}
