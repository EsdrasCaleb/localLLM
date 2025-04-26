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

public class Artists_getArtist_3_0_Test {

    @Mock
    private ArrayList<String> mockArrayList;

    @InjectMocks
    private Artists artists;

    @BeforeEach
    public void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.openMocks(this);
        Field field = Artists.class.getDeclaredField("artists");
        field.setAccessible(true);
        field.set(artists, mockArrayList);
    }

    @Test
    public void testGetArtist_IndexWithinBounds() {
        // Arrange
        String artistName = "John Doe";
        when(mockArrayList.get(0)).thenReturn(artistName);
        when(mockArrayList.size()).thenReturn(1);
        // Act
        String result = artists.getArtist(0);
        // Assert
        assertEquals(artistName, result);
    }

    @Test
    public void testGetArtist_IndexOutOfBounds() {
        // Arrange
        when(mockArrayList.size()).thenReturn(1);
        // Act
        String result = artists.getArtist(1);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetArtist_EmptyList() {
        // Arrange
        when(mockArrayList.size()).thenReturn(0);
        // Act
        String result = artists.getArtist(0);
        // Assert
        assertNull(result);
    }
}
