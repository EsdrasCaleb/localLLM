package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Directors_getDirector_3_0_Test {

    @InjectMocks
    private Directors directors;

    @Mock
    private ArrayList<String> mockDirectors;

    @BeforeEach
    public void setUp() {
        directors = new Directors();
        directors.setDirector(new String[] { "Director1", "Director2", "Director3" });
    }

    @Test
    public void testGetDirectorValidIndex() {
        String result = directors.getDirector(1);
        assertEquals("Director2", result);
    }

    @Test
    public void testGetDirectorInvalidIndex() {
        String result = directors.getDirector(5);
        assertNull(result);
    }

    @Test
    public void testGetDirectorEmptyList() {
        directors.setDirector(new String[] {});
        String result = directors.getDirector(0);
        assertNull(result);
    }

    @Test
    public void testGetDirectorBoundaryIndex() {
        String result = directors.getDirector(2);
        assertEquals("Director3", result);
    }

    @Test
    public void testGetDirectorNegativeIndex() {
        String result = directors.getDirector(-1);
        assertNull(result);
    }
}
