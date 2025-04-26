package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Directors_getDirector_3_0_Test {

    private Directors directors;

    @BeforeEach
    void setUp() {
        directors = new Directors();
    }

    @Test
    void testGetDirector_NegativeIndex() {
        // Arrange
        String[] directorNames = { "Director A", "Director B", "Director C" };
        directors.setDirector(directorNames);
        // Act
        // Negative index
        String result = directors.getDirector(-1);
        // Assert
        assertNull(result);
    }
}
