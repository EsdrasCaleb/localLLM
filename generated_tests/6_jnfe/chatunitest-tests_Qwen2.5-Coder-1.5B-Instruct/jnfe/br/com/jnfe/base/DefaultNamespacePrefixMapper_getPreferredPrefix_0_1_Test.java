package br.com.jnfe.base;

import java.util.Arrays;
import java.util.List;
import static org.mockito.ArgumentMatchers.anyString;
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

    @Mock
    private Logger logger;

    @InjectMocks
    private DefaultNamespacePrefixMapper mapper;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetPreferredPrefixForNfeNamespace() {
        // Given
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = "nfe";
        boolean requirePrefix = true;
        // When
        String result = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        // Then
        assertEquals("nf", result);
    }

    @Test
    void testGetPreferredPrefixWithoutSuggestion() {
        // Given
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = null;
        boolean requirePrefix = true;
        // When
        String result = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        // Then
        assertEquals("nf", result);
    }

    @Test
    void testGetPreferredPrefixWithoutRequirePrefix() {
        // Given
        String namespaceUri = "http://www.portalfiscal.inf.br/nfe";
        String suggestion = null;
        boolean requirePrefix = false;
        // When
        String result = mapper.getPreferredPrefix(namespaceUri, suggestion, requirePrefix);
        // Then
        assertNull(result);
    }
}
