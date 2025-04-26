package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Artists_toString_4_0_Test {

    private Artists artists;

    @BeforeEach
    public void setUp() {
        artists = new Artists();
    }

    @Test
    public void testToString() {
        // Create a mock ArrayList and add some elements to it
        ArrayList<String> mockArrayList = mock(ArrayList.class);
        when(mockArrayList.size()).thenReturn(3);
        when(mockArrayList.get(0)).thenReturn("Artist 1");
        when(mockArrayList.get(1)).thenReturn("Artist 2");
        when(mockArrayList.get(2)).thenReturn("Artist 3");
        // Use reflection to invoke the private setArtists method
        try {
            artists.getClass().getDeclaredMethod("setArtists", ArrayList.class).setAccessible(true);
            artists.getClass().getDeclaredMethod("setArtists", ArrayList.class).invoke(artists, mockArrayList);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Call the toString() method and assert the expected output
        String expectedOutput = "# of Lists = 3\n" + "artist - Artist 1\n" + "artist - Artist 2\n" + "artist - Artist 3\n";
        assertEquals(expectedOutput, artists.toString());
    }

    @Test
    public void testToStringNullList() {
        // Use reflection to invoke the private setArtists method with null
        try {
            artists.getClass().getDeclaredMethod("setArtists", ArrayList.class).setAccessible(true);
            artists.getClass().getDeclaredMethod("setArtists", ArrayList.class).invoke(artists, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Call the toString() method and assert the expected output
        String expectedOutput = "artists is null or size 0 \n";
        assertEquals(expectedOutput, artists.toString());
    }

    @Test
    public void testToStringEmptyList() {
        // Create an empty ArrayList and use reflection to invoke the private setArtists method
        ArrayList<String> emptyArrayList = new ArrayList<>();
        try {
            artists.getClass().getDeclaredMethod("setArtists", ArrayList.class).setAccessible(true);
            artists.getClass().getDeclaredMethod("setArtists", ArrayList.class).invoke(artists, emptyArrayList);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Call the toString() method and assert the expected output
        String expectedOutput = "artists is null or size 0 \n";
        assertEquals(expectedOutput, artists.toString());
    }
}
