package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Directors_toString_4_0_Test {

    private Directors directors;

    @BeforeEach
    void setUp() {
        directors = new Directors();
    }

    @Test
    void testToString_emptyDirectors() {
        assertEquals("Director is null or size 0\n", directors.toString());
    }

    @Test
    void testToString_nonEmptyDirectors() {
        String[] directorsArray = { "Director 1", "Director 2", "Director 3" };
        directors.setDirector(directorsArray);
        String expectedOutput = "# of Directors = 3\n" + "Director - Director 1\n" + "Director - Director 2\n" + "Director - Director 3\n";
        assertEquals(expectedOutput, directors.toString());
    }

    @Test
    void testToString_singleDirector() {
        String[] directorsArray = { "Director 1" };
        directors.setDirector(directorsArray);
        String expectedOutput = "# of Directors = 1\n" + "Director - Director 1\n";
        assertEquals(expectedOutput, directors.toString());
    }
}
