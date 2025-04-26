package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_get_6_4_Test {

    @Test
    public void testGet() {
        // Arrange
        Population population = new Population();
        List<Genome> genomes = new ArrayList<>();
        genomes.add(new Genome());
        population.genoms = genomes;
        int index = 1;
        Genome expectedGenome = genomes.get(index - 1);
        // Act
        Genome actualGenome = population.get(index);
        // Assert
        assertEquals(expectedGenome, actualGenome);
    }
}
