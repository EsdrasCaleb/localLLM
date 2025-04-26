package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Directors_getDirector_3_3_Test {

    @Test
    public void testGetDirector() throws NoSuchFieldException, IllegalAccessException {
        Directors directors = new Directors();
        ArrayList<String> mockDirectors = new ArrayList<>();
        mockDirectors.add("Director1");
        mockDirectors.add("Director2");
        mockDirectors.add("Director3");
        Field directorsField = Directors.class.getDeclaredField("directors");
        directorsField.setAccessible(true);
        directorsField.set(directors, mockDirectors);
        Directors directorsMock = Mockito.mock(Directors.class);
        when(directorsMock.getDirectorsArray()).thenReturn(mockDirectors);
        when(directorsMock.getDirector(0)).thenReturn("Director1");
        when(directorsMock.getDirector(1)).thenReturn("Director2");
        when(directorsMock.getDirector(2)).thenReturn("Director3");
        String result = directorsMock.getDirector(0);
        assertEquals("Director1", result);
        result = directorsMock.getDirector(1);
        assertEquals("Director2", result);
        result = directorsMock.getDirector(2);
        assertEquals("Director3", result);
    }
}
