package net.kencochrane.a4j.beans;

import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Directors_getDirector_3_0_Test {

    @Mock
    private Directors directors;

    @Test
    public void testGetDirectorWithinBounds() {
        ArrayList<String> directorsList = new ArrayList<>(Arrays.asList("Director1", "Director2", "Director3"));
        doReturn(directorsList).when(directors).getDirectorsArray();
        String result = directors.getDirector(1);
        assertEquals("Director2", result);
    }

    @Test
    public void testGetDirectorOutOfBounds() {
        ArrayList<String> directorsList = new ArrayList<>(Arrays.asList("Director1", "Director2", "Director3"));
        doReturn(directorsList).when(directors).getDirectorsArray();
        String result = directors.getDirector(5);
        assertNull(result);
    }
}
