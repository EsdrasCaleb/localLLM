package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_3_Test {

    private VectorGenome vectorGenome;

    private Vector genesMock;

    private Evaluator evaluatorMock;

    @BeforeEach
    public void setUp() {
        genesMock = mock(Vector.class);
        evaluatorMock = mock(Evaluator.class);
        vectorGenome = new VectorGenome(genesMock, evaluatorMock);
    }

    @Test
    public void testGetGene() throws Exception {
        // Given
        int index = 0;
        Object expectedValue = "someValue";
        when(genesMock.get(index)).thenReturn(expectedValue);
        // When
        Object actualValue = vectorGenome.getGene(index);
        // Then
        assertEquals(expectedValue, actualValue);
        verify(genesMock).get(index);
    }
}
