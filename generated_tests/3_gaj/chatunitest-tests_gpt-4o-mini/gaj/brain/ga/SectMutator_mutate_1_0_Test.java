package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class SectMutator_mutate_1_0_Test {

    private SectMutator sectMutator;

    private GAEnumAllelesSet allelesSet;

    private VectorGenome genome;

    @BeforeEach
    void setUp() {
        sectMutator = new SectMutator();
        allelesSet = mock(GAEnumAllelesSet.class);
        genome = mock(VectorGenome.class);
        sectMutator.setAllelesSet(allelesSet);
    }

    @Test
    void testMutate_NoMutation() {
        when(genome.getGenesCount()).thenReturn(5);
        when(allelesSet.allele()).thenReturn("newAllele");
        // Mock Random to always return a value greater than pmut
        setPrivateField(sectMutator, "rnd", new Random() {

            @Override
            public double nextDouble() {
                // Always greater than pmut = 0.5
                return 0.9;
            }
        });
        int result = sectMutator.mutate(genome, 0.5);
        verify(genome, never()).setGene(anyInt(), any());
        assertEquals(0, result);
    }

    @Test
    void testMutate_AllMutate() {
        when(genome.getGenesCount()).thenReturn(5);
        when(allelesSet.allele()).thenReturn("newAllele");
        // Mock Random to always return a value less than or equal to pmut
        setPrivateField(sectMutator, "rnd", new Random() {

            @Override
            public double nextDouble() {
                // Always less than pmut = 0.5
                return 0.1;
            }
        });
        int result = sectMutator.mutate(genome, 0.5);
        for (int i = 0; i < 5; i++) {
            verify(genome).setGene(i, "newAllele");
        }
        assertEquals(0, result);
    }

    @Test
    void testMutate_SomeMutate() {
        when(genome.getGenesCount()).thenReturn(5);
        when(allelesSet.allele()).thenReturn("newAllele");
        // Mock Random to return a mix of values
        setPrivateField(sectMutator, "rnd", new Random() {

            private int callCount = 0;

            @Override
            public double nextDouble() {
                // Mutate on even calls
                return callCount++ % 2 == 0 ? 0.1 : 0.9;
            }
        });
        int result = sectMutator.mutate(genome, 0.5);
        verify(genome, times(3)).setGene(anyInt(), eq("newAllele"));
        assertEquals(0, result);
    }

    private void setPrivateField(Object obj, String fieldName, Object value) {
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
