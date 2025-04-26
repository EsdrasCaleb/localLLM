package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_get_6_0_Test {

    @Test
    public void testGet() {
        // Arrange
        Population population = new Population();
        List<Genome> genomes = new ArrayList<>();
        genomes.add(new Genome());
        population.genoms = genomes;
        int index = 1;
        // Act
        Genome result = population.get(index);
        // Assert
        assertEquals(genomes.get(0), result);
    }
}
