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

    @InjectMocks
    private Directors directors;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWithDirectors() {
        ArrayList<String> mockDirectors = new ArrayList<>(Arrays.asList("Director1", "Director2"));
        directors.setDirector(mockDirectors.toArray(new String[0]));
        String expected = "# of Directors = 2\nDirector - Director1\nDirector - Director2\n";
        String result = directors.toString();
        assertEquals(expected, result);
    }

    @Test
    void testToStringWithEmptyDirectors() {
        directors.setDirector(new String[0]);
        String expected = "Director is null or size 0\n";
        String result = directors.toString();
        assertEquals(expected, result);
    }
}
