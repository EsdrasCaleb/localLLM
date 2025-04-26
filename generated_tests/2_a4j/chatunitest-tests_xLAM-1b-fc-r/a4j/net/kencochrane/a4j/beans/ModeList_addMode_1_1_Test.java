package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ModeList_addMode_1_1_Test {

    private ModeList modeList;

    private Mode mode;

    @BeforeEach
    public void setup() {
        modeList = new ModeList();
        mode = new Mode();
    }

    @Test
    @DisplayName("Testing addMode method")
    public void testAddMode() {
        modeList.addMode(mode);
        ArrayList<Mode> modes = modeList.getAllModes();
        assertEquals(mode, modes.get(0));
    }
}
