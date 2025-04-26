package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Artists_toString_4_0_Test {

    @Test
    void testToString_NullArtists() {
        Artists artists = new Artists();
        assertEquals("artists is null or size 0 \n", artists.toString());
    }

    @Test
    void testToString_EmptyArtists() {
        Artists artists = new Artists();
        try {
            Field artistsField = Artists.class.getDeclaredField("artists");
            artistsField.setAccessible(true);
            artistsField.set(artists, new ArrayList<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access artists field: " + e.getMessage());
        }
        assertEquals("artists is null or size 0 \n", artists.toString());
    }

    @Test
    void testToString_SingleArtist() {
        Artists artists = new Artists();
        try {
            Field artistsField = Artists.class.getDeclaredField("artists");
            artistsField.setAccessible(true);
            artistsField.set(artists, new ArrayList<>(Arrays.asList("Picasso")));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access artists field: " + e.getMessage());
        }
        assertEquals("# of Lists = 1\nartist - Picasso\n", artists.toString());
    }

    @Test
    void testToString_MultipleArtists() {
        Artists artists = new Artists();
        try {
            Field artistsField = Artists.class.getDeclaredField("artists");
            artistsField.setAccessible(true);
            artistsField.set(artists, new ArrayList<>(Arrays.asList("Picasso", "Monet", "Van Gogh")));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access artists field: " + e.getMessage());
        }
        assertEquals("# of Lists = 3\nartist - Picasso\nartist - Monet\nartist - Van Gogh\n", artists.toString());
    }

    @Test
    void testToString_ArtistsWithNullArtist() {
        Artists artists = new Artists();
        ArrayList<String> artistList = new ArrayList<>();
        artistList.add("Picasso");
        artistList.add(null);
        artistList.add("Van Gogh");
        try {
            Field artistsField = Artists.class.getDeclaredField("artists");
            artistsField.setAccessible(true);
            artistsField.set(artists, artistList);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access artists field: " + e.getMessage());
        }
        assertEquals("# of Lists = 3\nartist - Picasso\nartist - null\nartist - Van Gogh\n", artists.toString());
    }
}
