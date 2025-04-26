package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class Population_selectNextGenome_1_3_Test {

    @Mock
    Selector selector;

    @InjectMocks
    Population population;

    @Test
    void selectNextGenome_ShouldReturnNewGenome() {
        // Arrange
        population.setSelector(selector);
        // Act
        Genome selectedGenome = population.selectNextGenome();
        // Assert
        assertNotNull(selectedGenome);
    }
}
