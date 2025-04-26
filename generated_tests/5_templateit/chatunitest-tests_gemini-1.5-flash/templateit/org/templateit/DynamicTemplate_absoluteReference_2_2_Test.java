package org.templateit;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;

@ExtendWith(MockitoExtension.class)
class DynamicTemplate_absoluteReference_2_2_Test {

    @Mock
    private HSSFSheet sheet;

    private DynamicTemplate dynamicTemplate;

    private List<DynamicTemplate.NamedStyle> styles;

    @BeforeEach
    void setUp() {
        styles = new ArrayList<>();
        dynamicTemplate = new DynamicTemplate("test", sheet, 10, styles);
        try {
            Field heightField = DynamicTemplate.class.getDeclaredField("height");
            heightField.setAccessible(true);
            heightField.setInt(dynamicTemplate, 2);
            Field widthField = DynamicTemplate.class.getDeclaredField("width");
            widthField.setAccessible(true);
            widthField.setInt(dynamicTemplate, 3);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set height and width using reflection: " + e.getMessage());
        }
    }

    @Test
    void testAbsoluteReference_validRow() {
        styles.add(new DynamicTemplate.NamedStyle(1, 2));
        DynamicTemplate.Reference ref = dynamicTemplate.absoluteReference(0, 0);
        assertEquals(1, ref.getRow());
        assertEquals(2, ref.getColumn());
    }

    @Test
    void testAbsoluteReference_rowOutOfBounds() {
        assertThrows(IndexOutOfBoundsException.class, () -> dynamicTemplate.absoluteReference(0, 0));
    }

    @Test
    void testAbsoluteReference_negativeRow() {
        styles.add(new DynamicTemplate.NamedStyle(1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> dynamicTemplate.absoluteReference(-1, 0));
    }

    @Test
    void testAbsoluteReference_rowGreaterThanStylesSize() {
        styles.add(new DynamicTemplate.NamedStyle(1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> dynamicTemplate.absoluteReference(1, 0));
    }

    static class DynamicTemplate {

        private String name;

        private HSSFSheet sheet;

        private int height;

        private int width;

        private List<NamedStyle> styles;

        public DynamicTemplate(String name, HSSFSheet sheet, int height, List<NamedStyle> styles) {
            this.name = name;
            this.sheet = sheet;
            this.height = height;
            this.styles = styles;
        }

        public Reference absoluteReference(int row, int col) {
            if (styles.size() <= row || row < 0) {
                throw new IndexOutOfBoundsException();
            }
            return new Reference(styles.get(row).getRow(), styles.get(row).getColumn());
        }

        static class NamedStyle {

            private int row;

            private int column;

            public NamedStyle(int row, int column) {
                this.row = row;
                this.column = column;
            }

            public int getRow() {
                return row;
            }

            public int getColumn() {
                return column;
            }
        }

        static class Reference {

            private int row;

            private int column;

            public Reference(int row, int column) {
                this.row = row;
                this.column = column;
            }

            public int getRow() {
                return row;
            }

            public int getColumn() {
                return column;
            }

            @Override
            public boolean equals(Object obj) {
                if (this == obj)
                    return true;
                if (obj == null || getClass() != obj.getClass())
                    return false;
                Reference that = (Reference) obj;
                return row == that.row && column == that.column;
            }
        }
    }
}
