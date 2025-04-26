package br.com.jnfe.base;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sun.xml.bind.marshaller.NamespacePrefixMapper;

@ExtendWith(MockitoExtension.class)
public class DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test {

    @InjectMocks
    private DefaultNamespacePrefixMapper mapper;

    @Test
    public void testGetPreferredPrefix_NamespaceUriMatches_ReturnsPreferredPrefix() {
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = null;
        boolean requirePrefix = false;
        String expectedPrefix = "nf";
        String actualPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals(expectedPrefix, actualPrefix);
    }

    @Test
    public void testGetPreferredPrefix_NamespaceUriDoesNotMatch_ReturnsNull() {
        String namespaceUri = "http://example.com";
        String suggestion = null;
        boolean requirePrefix = false;
        String expectedPrefix = null;
        String actualPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals(expectedPrefix, actualPrefix);
    }

    @Test
    public void testGetPreferredPrefix_RequirePrefixTrue_ReturnsNull() {
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = null;
        boolean requirePrefix = true;
        String expectedPrefix = null;
        String actualPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals(expectedPrefix, actualPrefix);
    }

    @Test
    public void testGetPreferredPrefix_RequirePrefixFalse_SuggestionProvided_ReturnsSuggestion() {
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = "customPrefix";
        boolean requirePrefix = false;
        String expectedPrefix = suggestion;
        String actualPrefix = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        assertEquals(expectedPrefix, actualPrefix);
    }
}
