package net.kencochrane.a4j.file;

import java.io.File;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_getASINFile_4_0_Test {

    @Test
    public void testGetASINFile() throws IOException {
        FileUtil fileUtil = Mockito.mock(FileUtil.class);
        Mockito.when(fileUtil.getASINFile("123456789", "type", "offer", "page")).thenReturn(new File("C:\\cache\\123456789_type_offer_page.xml"));
        File result = fileUtil.getASINFile("123456789", "type", "offer", "page");
        Mockito.verify(fileUtil).getASINFile("123456789", "type", "offer", "page");
        assert result != null;
    }
}
